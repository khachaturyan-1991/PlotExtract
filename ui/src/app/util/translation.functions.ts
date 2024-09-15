import { LOCATION_INITIALIZED } from "@angular/common";
import { HttpClient } from "@angular/common/http";
import { Injector } from "@angular/core";
import { TranslateService } from "@ngx-translate/core";
import { TranslateHttpLoader } from "@ngx-translate/http-loader";
import { StateService } from "../_service/state.service";

export function createTranslateLoader(http: HttpClient) {
  return new TranslateHttpLoader(http, './assets/i18n/', '.json');
}

export function appInitializerFactory(translate: TranslateService, injector: Injector, stateService: StateService) {
  return () => new Promise<any>((resolve: any) => {
    const locationInitialized = injector.get(LOCATION_INITIALIZED, Promise.resolve(null));
    locationInitialized.then(() => {
      stateService.getLanguage().subscribe(
        response => {
          const langToSet = response.language;
          translate.setDefaultLang('en');
          translate.use(langToSet).subscribe(() => {
            console.info(`Successfully initialized '${langToSet}' language.'`);
          }, err => {
            console.error(`Problem with '${langToSet}' language initialization.'`);
          }, () => {
            resolve(null);
          });
      });
      resolve(null);
    });
  });
}
